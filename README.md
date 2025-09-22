Troubleshooting
===========
1. Problem during plotting data. Error message: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown plt.show()
    Solution 1: Use plt.savefig('output.png') instead of plt.show() to save the plot as an image file.
    Solution 2: If you are using WSL2 Ubuntu on Windows try to install python3-tk package:
        sudo apt-get install python3-tk
