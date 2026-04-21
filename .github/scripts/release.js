module.exports = async ({ github, context, core }) => {
  const { VERSION, FILE_NAME, FILE_TYPE } = process.env;
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  const fs = require("fs");
  const path = require("path");

  let release;

  found: for await (const res of github.paginate.iterator(
    github.rest.repos.listReleases,
    {
      owner,
      repo,
      per_page: 100,
    },
  )) {
    for (const rel of res.data) {
      if (rel.tag_name === VERSION) {
        release = rel;
        break found;
      }
    }
  }

  if (!release) {
    console.log(`Release ${VERSION} not found, creating...`);
    const created = await github.rest.repos.createRelease({
      owner,
      repo,
      tag_name: VERSION,
      name: VERSION,
      generate_release_notes: true,
      draft: VERSION.includes("-rc"),
    });
    release = created.data;
  }

  const listed = await github.rest.repos.listReleaseAssets({
    owner,
    repo,
    release_id: release.id,
  });
  const duplicate = listed.data.find((item) => item.name === FILE_NAME);
  if (duplicate) {
    await github.rest.repos.deleteReleaseAsset({
      owner,
      repo,
      asset_id: duplicate.id,
    });
  }

  const data = fs.readFileSync(path.join(process.cwd(), FILE_NAME));
  await github.rest.repos.uploadReleaseAsset({
    headers: {
      "content-type": FILE_TYPE,
      "content-length": data.length,
    },
    owner,
    repo,
    release_id: release.id,
    name: FILE_NAME,
    data,
  });

  core.info(`Release ${VERSION} is ready: ${release.html_url}`);
};
