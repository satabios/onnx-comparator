# ONNX Model Comparator for VS Code

[![Visual Studio Marketplace Version](https://img.shields.io/visual-studio-marketplace/v/satabios.onnx-model-comparator)](https://marketplace.visualstudio.com/items?itemName=satabios.onnx-model-comparator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compare two ONNX (Open Neural Network Exchange) models side-by-side within Visual Studio Code and visualize their differences.

You can perceive this as a beyond compare for ONNX models.

![Overview](images/recording.gif)

## Features

*   **Select Models:** Easily select two `.onnx` model files from your workspace using file pickers.
*   **Side-by-Side View:** Displays comparison results in a dedicated webview panel.
*   **Compare Key Components:**
    *   **Model Information:** Compares metadata like IR version, producer, model version.
    *   **Inputs:** Shows differences in input names, data types, and shapes.
    *   **Outputs:** Shows differences in output names, data types, and shapes.
    *   **Nodes (Operators):** Lists added, removed, or modified nodes, detailing changes in inputs, outputs, attributes, and parameter shapes.
    *   **Initializers:** Compares weights and constants, highlighting differences in data type and shape.
*   **Highlight Differences:** Uses color-coding (green for added, red for removed, orange for modified) to easily spot changes.
*   **Node Search:** Filter the nodes table by name to quickly find specific operators.

## Usage

1.  Open the VS Code Command Palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Windows/Linux).
2.  Run the command: `Compare ONNX Models`.
3.  You will be prompted to select the first ONNX model file.
4.  You will then be prompted to select the second ONNX model file.
5.  A new panel will open displaying the comparison results.

## Requirements

*   Visual Studio Code version 1.85.0 or higher.
*   Node.js (for running the extension host).

## Known Issues

*   Comparison of large initializers (weights) currently only checks for existence, data type, and shape differences, not value differences.
*   Display of very complex tensor shapes or attributes might be simplified.

## License

This extension is licensed under the [MIT License](LICENSE.txt).
