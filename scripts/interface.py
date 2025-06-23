import sys
import os

# sys.path.append('.')

from PyQt5.QtWidgets import QApplication

from hierulz.interface import InterfaceHClassification




def main():
    app = QApplication(sys.argv)
    interface = InterfaceHClassification()
    interface.show()  # Show the interface window
    sys.exit(app.exec_())  # Start the application event loop

if __name__ == "__main__":

    # Initialize the interface with the dataset
    main()
    
    