module.exports = async ({ github, context, core }) => {
  const { VERSION, FILES, HTP_SIGNED, LLAMA_SHA } = process.env;
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  const fs = require("fs");
  const path = require("path");

  const MAX_RETRIES = 2;
  const RETRY_DELAY_MS = 5000;

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function withRetry(fn, label) {
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        return await fn();
      } catch (err) {
        if (attempt === MAX_RETRIES) throw err;
        core.warning(
          `${label}: attempt ${attempt}/${MAX_RETRIES} failed (${err.message}), retrying in ${RETRY_DELAY_MS}ms...`,
        );
        await sleep(RETRY_DELAY_MS * attempt);
      }
    }
  }

  // Any SemVer pre-release (e.g. v1.2.3-rc.1, v1.2.3-alpha.1) is published as draft.
  const shouldPublish = !VERSION.includes("-");

  const htpNote =
    HTP_SIGNED === "true"
      ? `## Hexagon HTP\n\nMicrosoft-signed HTP catalog (llama.cpp @ \`${LLAMA_SHA}\`). No cert import required on Windows on Snapdragon.`
      : `## Hexagon HTP\n\nSelf-signed HTP catalog (llama.cpp @ \`${LLAMA_SHA}\`). End users must enable test signing and import \`ggml-htp-v1.cer\` per [notes/run.md](../blob/${VERSION}/notes/run.md).\n\nOperators: ship \`libggml-htp-to-sign-${LLAMA_SHA}.zip\` to the signing pipeline, then upload the signed result as \`libggml-htp-${LLAMA_SHA}.zip\` to \`s3://qaihub-public-assets/llama-cpp/\` and re-run this release.`;

  let release;
  for await (const res of github.paginate.iterator(
    github.rest.repos.listReleases,
    { owner, repo, per_page: 100 },
  )) {
    release = res.data.find((r) => r.tag_name === VERSION);
    if (release) break;
  }

  // Merge htpNote into the release body, replacing any stale HTP section from a
  // prior run (e.g. self-signed → Microsoft-signed once the S3 bundle lands).
  const mergeHtpNote = (body) => {
    const base = (body || "").replace(/## Hexagon HTP[\s\S]*$/m, "").trimEnd();
    return base ? `${base}\n\n${htpNote}` : htpNote;
  };

  async function createDraftRelease(body) {
    const created = await github.rest.repos.createRelease({
      owner,
      repo,
      tag_name: VERSION,
      name: VERSION,
      body,
      generate_release_notes: true,
      draft: true,
    });
    return created.data;
  }

  if (!release) {
    core.info(`Release ${VERSION} not found, creating as draft...`);
    release = await createDraftRelease(htpNote);
  } else {
    // If a prior run published the release, revert to draft so we can upload
    // assets. GitHub's "immutable releases" feature may block this update — in
    // that case delete the existing release and recreate as a fresh draft.
    const needsDraft = !release.draft;
    const body = mergeHtpNote(release.body);
    if (needsDraft || body !== release.body) {
      core.info(
        needsDraft
          ? `Release ${VERSION} is published; reverting to draft for asset upload...`
          : `Updating release body...`,
      );
      try {
        const patched = await github.rest.repos.updateRelease({
          owner,
          repo,
          release_id: release.id,
          body,
          draft: true,
        });
        release = patched.data;
      } catch (err) {
        // 422 = immutable release — cannot revert to draft. Delete and recreate.
        if (err.status === 422) {
          core.warning(
            `Cannot revert release to draft (immutable). Deleting and recreating...`,
          );
          await github.rest.repos.deleteRelease({
            owner,
            repo,
            release_id: release.id,
          });
          release = await createDraftRelease(body);
        } else {
          throw err;
        }
      }
    }
  }

  const existingAssets = (
    await github.rest.repos.listReleaseAssets({
      owner,
      repo,
      release_id: release.id,
    })
  ).data;

  const files = FILES.split("\n").map((f) => f.trim()).filter(Boolean);
  const uploaded = [];
  const failed = [];

  for (const fileName of files) {
    try {
      const duplicate = existingAssets.find((a) => a.name === fileName);
      if (duplicate) {
        await withRetry(
          () =>
            github.rest.repos.deleteReleaseAsset({
              owner,
              repo,
              asset_id: duplicate.id,
            }),
          `Delete existing ${fileName}`,
        );
      }
      const data = fs.readFileSync(path.join(process.cwd(), fileName));
      const contentType =
        fileName.endsWith(".txt") || fileName.endsWith(".sha256")
          ? "text/plain"
          : fileName.endsWith(".exe")
            ? "application/octet-stream"
            : fileName.endsWith(".tar.gz") || fileName.endsWith(".tgz")
              ? "application/gzip"
              : fileName.endsWith(".aar")
                ? "application/java-archive"
                : fileName.endsWith(".apk")
                  ? "application/vnd.android.package-archive"
                  : "application/zip";
      await withRetry(
        () =>
          github.rest.repos.uploadReleaseAsset({
            owner,
            repo,
            release_id: release.id,
            name: fileName,
            data,
            headers: {
              "content-type": contentType,
              "content-length": data.length,
            },
          }),
        `Upload ${fileName}`,
      );
      uploaded.push(fileName);
      core.info(`Uploaded ${fileName}`);
    } catch (err) {
      failed.push({ file: fileName, error: err.message });
      core.error(`Failed to upload ${fileName}: ${err.message}`);
    }
  }

  // Publish the release (remove draft status) regardless of upload failures —
  // ship whatever assets succeeded rather than blocking the entire release.
  if (shouldPublish) {
    core.info(`Publishing release ${VERSION} (${uploaded.length}/${files.length} assets)...`);
    const published = await github.rest.repos.updateRelease({
      owner,
      repo,
      release_id: release.id,
      draft: false,
    });
    release = published.data;
  }

  core.info(`Release ${VERSION} is ready: ${release.html_url}`);

  if (failed.length > 0) {
    const summary = failed.map((f) => `  - ${f.file}: ${f.error}`).join("\n");
    core.setFailed(
      `${failed.length}/${files.length} asset(s) failed to upload:\n${summary}`,
    );
  }
};
