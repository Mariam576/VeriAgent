# Putting VeriAgent on the internet (Render + Neon)

This guide takes the VeriAgent that runs on your PC and gives it a public address
such as `https://veriagent-xxxx.onrender.com`, on free plans:

| Part | Where it runs | What it is |
|---|---|---|
| The application (API and interface, one container) | [Render](https://render.com) | a host that builds the `Dockerfile` of your GitHub repository and runs it |
| The database (PostgreSQL with pgvector) | [Neon](https://neon.com) | hosted PostgreSQL |
| The language model and the embeddings | Gemini API | the same key you already use |

You need a GitHub account with the VeriAgent repository pushed to it, and about
half an hour.

> **Honest note.** The pieces this guide relies on are tested in this project: reading
> a hosted database's connection string, the small image's settings, the start-up
> checks, HTTPS behind a proxy. The deployment as a whole has **not** been run on Render
> by the author of this guide, because that needs your accounts. If a step behaves
> differently from what is written here, the "If something goes wrong" section is the
> place to look, and Render's log shows the reason in plain words.

## What "free" means here

Free plans change. These were the terms when this guide was written (October 2026);
check the two pricing pages before you rely on them.

**Render, free web service**

- 512 MB of memory and a small share of a CPU.
- **It goes to sleep after 15 minutes without a visitor.** The next visitor waits
  up to about a minute while it starts again. Nothing is lost: everything that
  matters is in the database.
- No permanent disk. Files written by the application disappear at every restart.
  VeriAgent is configured for this (`KEEP_UPLOADED_FILES=false`, see below).

**Neon, free plan**

- A limited amount of storage per project (in the order of a gigabyte or less) and
  a monthly allowance of compute time. A few hundred documents fit comfortably.
- **The database also goes to sleep**, after 5 minutes without a query, and wakes
  in about a second. VeriAgent expects that: it replaces connections that went
  stale and waits for the database at start-up.

**Gemini API, free tier**

- A limited number of requests per minute and per day. `LLM_REQUESTS_PER_MINUTE=10`
  and `MAX_INVESTIGATIONS_PER_DAY=25` keep VeriAgent inside it.
- On the free tier Google may use what you send to improve its products. **Do not
  upload confidential documents** to a deployment that uses a free-tier key.

What you give up compared with your PC: speed after a pause, and local embeddings.
PyTorch alone needs more memory than the free plan has, so the hosted version asks
the Gemini API for embeddings instead. The application without PyTorch was measured
at about 200 MB.

## Step 1: the database on Neon

1. Sign up at <https://neon.com> and create a **project**. Give it a name
   (`veriagent`), keep the suggested PostgreSQL version, and choose the **region**
   nearest to you. Remember the region: the application should run near it.
2. Open **Connect** on the project page. It shows a connection string like

   ```
   postgresql://neondb_owner:xxxxxxxx@ep-cool-name-123456.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
   ```

   Switch **Connection pooling off** in that dialog (the host name then has no
   `-pooler` in it) and copy the string. Both kinds work with VeriAgent; the direct
   one is the simpler of the two.
3. Keep the string for step 3. **It contains the database password.** Treat it like
   a password: do not paste it into a file of the repository, a chat or a screenshot.

You do not have to create tables or switch pgvector on. VeriAgent does both when
it starts for the first time.

## Step 2: choose a sign-up code

Your site will be reachable by anyone who knows its address. Without a sign-up
code, anyone could create an account and spend your Gemini allowance.

Think of a word or phrase (for example four random words). You type it once into
Render in the next step, and once when you create your own account. Give it only
to people you want to let in.

## Step 3: the application on Render

The file `render.yaml` in the repository describes the service: Render calls this
a **Blueprint**.

1. Sign up at <https://render.com> with your GitHub account, and allow Render to
   see the `VeriAgent` repository.
2. In the dashboard choose **New → Blueprint** and pick the repository. Render
   reads `render.yaml` and shows one service, `veriagent`.
3. Render asks for the three values that are not in the file:

   | It asks for | You enter |
   |---|---|
   | `DATABASE_URL` | the connection string from Neon |
   | `GEMINI_API_KEY` | your Gemini key |
   | `SIGNUP_CODE` | the code from step 2 |

4. Apply. Render builds the image and starts it. The first build takes several
   minutes (it installs Node and Python packages); later ones are faster.
5. When the service shows **Live**, open its address. You should see the
   VeriAgent sign-in page.

If you want a different region than Singapore, change `region:` in `render.yaml`
before step 2 and push the change. A region cannot be changed afterwards without
creating the service again.

### What the Blueprint sets, and why

| Setting | Value | Why |
|---|---|---|
| `LOCAL_EMBEDDINGS` | `false` | Render passes this to `docker build`; the `Dockerfile` then leaves PyTorch out |
| `EMBEDDING_PROVIDER`, `EMBEDDING_DIM` | `gemini`, `768` | embeddings from the API instead of a local model |
| `AUTH_MODE` | `jwt` | people have to sign in |
| `JWT_SECRET` | generated by Render | signs the sign-in tokens; never leaves Render |
| `TRUST_PROXY_HEADERS` | `true` | Render's proxy tells the app that the visitor used HTTPS, so the sign-in cookie is marked "HTTPS only" |
| `CLIENT_ADDRESS_HEADER` | `cf-connecting-ip` | the visitor's address for the per-address limits, from a header the network in front of Render sets itself |
| `KEEP_UPLOADED_FILES` | `false` | there is no permanent disk; the text, the vectors and the inspection result of a document are stored in the database, the file itself is deleted after processing |
| `DB_POOL_SIZE`, `DB_MAX_OVERFLOW` | `3`, `2` | a small database allows few connections |
| `MAX_BACKGROUND_JOBS` | `1` | one document is processed at a time, to stay inside the memory |
| `ACTION_SERVER` | `tickets` | follow-up issues go to the built-in ticket list (see "Optional" below) |

`EMBEDDING_DIM` is fixed by the first start: the database column is created with
that size. Do not change it later.

## Step 4: your account

1. On the sign-in page choose **Create account**, and enter your name, email
   address, a password and the sign-up code.
2. Upload a document, ask a question.
3. When everyone who should have an account has one, you can close the door:
   in Render open the service → **Environment**, add `ALLOW_REGISTRATION` with the
   value `false`, and save. People with an account can still sign in.

VeriAgent sends no emails. There is no "forgot my password" link: someone who
forgets a password signs in with Google (if you switched that on) or asks you.

## Optional: sign in with Google

1. In the [Google Cloud console](https://console.cloud.google.com/apis/credentials)
   create an **OAuth client ID** of the type **Web application**.
2. Under **Authorised JavaScript origins** add the address of your site exactly as
   the browser shows it, without a path: `https://veriagent-xxxx.onrender.com`.
   (For your PC add `http://localhost:8000` as well.)
3. Copy the client id (it ends in `.apps.googleusercontent.com`; it is not a secret)
   and add it in Render → **Environment** as `GOOGLE_CLIENT_ID`.

The sign-in page then shows a Google button. The sign-up code applies here too:
a new person types it into the "Create account" tab before pressing the button.

## Optional: real GitHub issues instead of the ticket list

On the free plan the ticket list is a file on a disk that is emptied at every
restart. VeriAgent still remembers in the database *that* an issue was created
and under which number, but the ticket itself is gone. For a record that lasts,
let VeriAgent create issues in a GitHub repository:

1. Create a **fine-grained personal access token** at
   <https://github.com/settings/personal-access-tokens>: access to **only** the one
   repository that should receive the issues, permission **Issues: Read and write**,
   nothing else.
2. In Render → **Environment** set `ACTION_SERVER` to `github`, and add
   `GITHUB_REPOSITORY` (`owner/repository`) and `GITHUB_TOKEN`.

An issue is only ever created after a person approved exactly that issue in the
interface. This path talks to GitHub's own MCP server; in this project it was
tested against a stand-in with the same tool, not against GitHub itself, so try
it once with a test repository first.

## Updating

Push to the `main` branch on GitHub. Render notices, builds the new version and
switches over when its `/health` check answers. Database changes are applied by
the application at start-up (`app/database/migrations.py`).

Values that are written in `render.yaml` are owned by that file: change them
there and push. Values you added in the dashboard (the secrets, `GOOGLE_CLIENT_ID`,
`ALLOW_REGISTRATION`) stay as you set them.

## If something goes wrong

Open the service in Render and look at **Logs**. VeriAgent writes what it is doing
there, and removes keys and passwords from error messages before it logs them.

| What you see | What it means | What to do |
|---|---|---|
| `VeriAgent cannot start with these settings:` followed by a list | the start-up check (`app/startup.py`) found settings that cannot work together | each line says what to change |
| `Database not ready yet (attempt …)` fifteen times, then `Could not connect to the database` | the connection string is wrong, or the database was deleted | copy the string from Neon again into `DATABASE_URL` |
| `The database stores …-dimensional embeddings but EMBEDDING_DIM is …` | `EMBEDDING_DIM` no longer matches the database | set it back to the number the message names |
| The page takes a minute to appear | the service was asleep | nothing; it is awake now |
| "Too many requests" or a quota message from Gemini in a result | the free tier of the Gemini API is used up for the moment | wait; or lower `LLM_REQUESTS_PER_MINUTE` |
| A document stays "processing", then shows "failed" | the service restarted while it was being read | upload it again |
| The build stops with an out-of-memory error | the full image was built | check that `LOCAL_EMBEDDINGS` is `false` in the service's environment |
| Sign-in works, but a reload shows the sign-in page again | the sign-in cookie was not stored | open the site with `https://`, and check that `TRUST_PROXY_HEADERS` is `true` |
| The Google button does not appear, or shows an error | the origin is not registered | compare the address in step 2 of the Google section, character by character |

## Taking it down

Delete the service in Render (**Settings → Delete Web Service**) and the project in
Neon (**Settings → Delete**). Deleting the Neon project deletes every document,
investigation and account. A single person can also remove their own data without
that: the account button at the top right → **Delete account**.

## The same image anywhere else

Nothing in the application knows about Render. Any host that can run a container
and reach a PostgreSQL database with the pgvector extension will do:

```bash
docker build --build-arg LOCAL_EMBEDDINGS=false -t veriagent .
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://..." \
  -e GEMINI_API_KEY="..." \
  -e EMBEDDING_PROVIDER=gemini -e EMBEDDING_DIM=768 \
  -e SIGNUP_CODE="..." \
  veriagent
```

The container listens on the port named by the `PORT` variable (8000 if it is not
set). Behind a proxy that terminates HTTPS, also set `TRUST_PROXY_HEADERS=true`;
without one, leave it `false`, because a visitor could then fake the headers it
trusts.

## What happens between the browser and the database

```
browser ──HTTPS──▶ Render's proxy ──HTTP──▶ the VeriAgent container ──TLS──▶ Neon
                   (adds X-Forwarded-Proto)   uvicorn on $PORT               PostgreSQL + pgvector
                                              │
                                              └──HTTPS──▶ Gemini API (model and embeddings)
```

- **HTTPS ends at the proxy.** The container only ever sees plain HTTP. That is why
  the application has to be told to believe the proxy's `X-Forwarded-Proto` header
  (`TRUST_PROXY_HEADERS`) before it marks the sign-in cookie "HTTPS only" and sends
  the `Strict-Transport-Security` header.
- **The connection string is translated.** Neon writes it for the `libpq` library
  (`?sslmode=require&channel_binding=require`). The application uses two database
  drivers (asyncpg for its own tables, psycopg for LangGraph's checkpoints), and
  asyncpg does not understand those words. `app/database/urls.py` turns the one
  string into the right form for each, and switches prepared statements off when
  the address is a connection pooler.
- **One process.** The interface, the API, the background jobs and the hourly
  upkeep all run in the one container. That is what makes a free plan enough; it
  is also why the in-memory rate limits and job list are sufficient. With more
  than one container they would have to move into the database or Redis.
