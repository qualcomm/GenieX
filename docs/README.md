# Getting Started with Mintlify

## Prerequisites

Before you begin, make sure you have **Node.js 20 or later** installed.

<details>
<summary><strong>Install Node.js (includes npm)</strong></summary>

**macOS** (via Homebrew):

```bash
brew install node
```

**Windows** (via winget):

```bash
winget install OpenJS.NodeJS.LTS
```

After installing, verify with:

```bash
node -v
npm -v
```

</details>

## Install and Use Mintlify Locally

**1.** Install the Mintlify [CLI](https://www.npmjs.com/package/mint):

```bash
npm i -g mint
```

**2.** Navigate to the docs directory (where `docs.json` is located) and start the local preview:

```bash
mint dev
```

Alternatively, without installing the CLI globally:

```bash
npx mint dev
```

A local preview of your documentation will be available at `http://localhost:3000`.
Please note that the rendering behavior in the local preview may differ from the deployed version. The deployed result should be considered as the final reference.
