module.exports = async ({ github, context, core }) => {
  const { VERSION, FILES, HTP_SIGNED, LLAMA_SHA } = process.env;
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  const fs = require("fs");
  const path = require("path");

  // Any SemVer pre-release (e.g. v1.2.3-rc.1, v1.2.3-alpha.1) is published as draft.
  const isDraft = VERSION.includes("-");

  const htpNote =
    HTP_SIGNED === "true"
      ? `## Hexagon HTP\n\nMicrosoft-signed HTP catalog (llama.cpp @ \`${LLAMA_SHA}\`). No cert import required on Windows on Snapdragon.`
      : `## Hexagon HTP\n\nSelf-signed HTP catalog (llama.cpp @ \`${LLAMA_SHA}\`). End users must enable test signing and import \`ggml-htp-v1.cer\` per [docs/run.md](../blob/${VERSION}/docs/run.md).\n\nOperators: ship \`libggml-htp-to-sign-${LLAMA_SHA}.zip\` to the signing pipeline, then upload the signed result as \`libggml-htp-${LLAMA_SHA}.zip\` to \`s3://qaihub-public-assets/llama-cpp/\` and re-run this release.`;

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

  if (!release) {
    core.info(`Release ${VERSION} not found, creating (draft=${isDraft})...`);
    const created = await github.rest.repos.createRelease({
      owner,
      repo,
      tag_name: VERSION,
      name: VERSION,
      body: htpNote,
      generate_release_notes: true,
      draft: isDraft,
    });
    release = created.data;
  } else {
    const body = mergeHtpNote(release.body);
    if (body !== release.body) {
      const patched = await github.rest.repos.updateRelease({
        owner,
        repo,
        release_id: release.id,
        body,
      });
      release = patched.data;
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
  for (const fileName of files) {
    const duplicate = existingAssets.find((a) => a.name === fileName);
    if (duplicate) {
      await github.rest.repos.deleteReleaseAsset({
        owner,
        repo,
        asset_id: duplicate.id,
      });
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
              : "application/zip";
    await github.rest.repos.uploadReleaseAsset({
      owner,
      repo,
      release_id: release.id,
      name: fileName,
      data,
      headers: {
        "content-type": contentType,
        "content-length": data.length,
      },
    });
    core.info(`Uploaded ${fileName}`);
  }

  core.info(`Release ${VERSION} is ready: ${release.html_url}`);
};
