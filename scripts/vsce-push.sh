# Get version from package.json
VERSION=$(jq -r '.version' package.json)

git tag -a "v$VERSION" -m "Release version $VERSION"
git push origin "v$VERSION"