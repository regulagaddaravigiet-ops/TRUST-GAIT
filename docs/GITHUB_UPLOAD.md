# Publish code and the full archive

The full ZIP contains substantial synthetic validation checkpoints. Upload it as a GitHub **Release asset**, and upload extracted source/config/docs separately to the repository. Do not put the entire large ZIP into ordinary Git history.

GitHub blocks ordinary Git files larger than100MiB and browser file uploads larger than25MiB. Release assets or Git LFS are the appropriate mechanisms for large artifacts. Source: [GitHub large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), checked2026-09-18.

1. Create your repository on GitHub under the intended owner/name.
2. Extract the ZIP. The included `.gitignore` excludes validation checkpoint binaries and local data from ordinary commits.
3. Commit the code, configs, docs and reports. Set your own Git identity.
4. Create a release tag and upload the complete ZIP as a release attachment.
5. Add the actual repository/release URLs to README and citation metadata after publication.
6. Choose a license you have authority to grant. The archive does not invent permission for the manuscript, earlier code or third-party datasets.

Example commands (replace OWNER/REPO yourself):

```bash
git init
git add .
git commit -m "Add TriTrust implementation and validation documentation"
git branch -M main
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
git tag v2.0.0
git push origin v2.0.0
```

Then use GitHub Releases → Draft a new release → choose v2.0.0 → attach full ZIP. State explicitly: "Reconstructed implementation and synthetic software validation; manuscript-level historical experiments not reproduced."

Do not describe the synthetic weights as original paper checkpoints. Do not include restricted dataset images without permission. This package has not performed any GitHub upload on your behalf.
