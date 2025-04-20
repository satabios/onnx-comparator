# Remove unnecessary files
rm -rf dist
rm -rf node_modules
rm -rf .vscode-test
rm -rf *.vsix

#  npm install diff@5.1.0 onnxruntime-node@1.17.0 protobufjs@7.5.0

vsce package
code --install-extension *.vsix
# vscde publish

