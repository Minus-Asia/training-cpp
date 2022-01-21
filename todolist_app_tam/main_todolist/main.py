# This Python file uses the following encoding: utf-8
import sys
from os import path

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QDialog
from PyQt5.uic import loadUi


class todolist(QWidget):
    def __init__(self):
        super(todolist, self).__init__()
        self.load_ui()
        self.add_btn.clicked.connect(self.add_action_clicked)
        self.remove_btn.clicked.connect(self.remove_action_clicked)
        self.added_text = None

    def load_ui(self):
        ui_path = path.dirname(path.abspath(__file__))
        loadUi(path.join(ui_path, "form.ui"), self)

    def add_action_clicked(self):
        add_action_widget = add_action()
        add_action_widget.exec()
        self.added_text = add_action_widget.added_text
        self.listWidget.addItem(self.added_text)

    def remove_action_clicked(self):
        self.listWidget.takeItem(self.listWidget.currentRow())


class add_action(QDialog):
    def __init__(self):
        super(add_action, self).__init__()
        self.added_text = None
        self.load_ui()
        self.ok_btn.clicked.connect(self.ok_click)

    def load_ui(self):
        ui_path = path.dirname(path.abspath(__file__))
        loadUi(path.join(ui_path, "add_dialog.ui"), self)

    def ok_click(self):
        self.added_text = self.action_add.text()
        self.accept()


if __name__ == "__main__":
    app = QApplication([])
    widget = todolist()
    widget.show()
    sys.exit(app.exec_())
