# Remove unnecessary files
rm -rf dist
rm -rf node_modules
rm -rf .vscode-test
rm -rf *.vsix

vsce package
code --install-extension *.vsix


# Publish to marketplace
# vscde publish

