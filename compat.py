"""
Qt compatibility layer
======================
Always import Qt symbols from this module rather than from PyQt5 / PyQt6
directly.  PyMOL's ``pymol.Qt`` shim re-exports the correct binding for
whichever Qt version the running PyMOL was built against; this module adds
a second layer that normalises the API differences between Qt 5 and Qt 6.

Differences handled here
------------------------
- Enum scoping  (Qt6 requires ``Qt.AlignmentFlag.AlignLeft``;
                 Qt5 uses the flat ``Qt.AlignLeft``)
- ``QAction`` location  (QtWidgets in Qt5, QtGui in Qt6)
- ``QDialog.exec_()``   (renamed to ``exec()`` in Qt6;
                         use ``dialog.exec()`` everywhere — accepted by both)
"""

from __future__ import annotations

from pymol.Qt import QtCore, QtGui, QtWidgets  # noqa: F401  (re-exported)

# Persistent settings: exposed here so GUI code stays routed through pymol.Qt.
QSettings = QtCore.QSettings

# Detect Qt major version
QT_MAJOR: int = int(QtCore.QT_VERSION_STR.split(".")[0])

# Enum aliases — use these throughout the plugin instead of raw Qt enums
if QT_MAJOR >= 6:
    AlignLeft = QtCore.Qt.AlignmentFlag.AlignLeft
    AlignRight = QtCore.Qt.AlignmentFlag.AlignRight
    AlignCenter = QtCore.Qt.AlignmentFlag.AlignCenter
    AlignTop = QtCore.Qt.AlignmentFlag.AlignTop
    AlignBottom = QtCore.Qt.AlignmentFlag.AlignBottom
    AlignVCenter = QtCore.Qt.AlignmentFlag.AlignVCenter
    Horizontal = QtCore.Qt.Orientation.Horizontal
    Vertical = QtCore.Qt.Orientation.Vertical
    ScrollBarAlwaysOff = QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    ScrollBarAsNeeded = QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
else:
    AlignLeft = QtCore.Qt.AlignLeft
    AlignRight = QtCore.Qt.AlignRight
    AlignCenter = QtCore.Qt.AlignCenter
    AlignTop = QtCore.Qt.AlignTop
    AlignBottom = QtCore.Qt.AlignBottom
    AlignVCenter = QtCore.Qt.AlignVCenter
    Horizontal = QtCore.Qt.Horizontal
    Vertical = QtCore.Qt.Vertical
    ScrollBarAlwaysOff = QtCore.Qt.ScrollBarAlwaysOff
    ScrollBarAsNeeded = QtCore.Qt.ScrollBarAsNeeded

# QAction: lives in QtWidgets (Qt5) and moved to QtGui (Qt6)
if QT_MAJOR >= 6:
    QAction = QtGui.QAction
else:
    QAction = QtWidgets.QAction

# QFormLayout field-growth-policy enum
# Qt5: QFormLayout.AllNonFixedFieldsGrow  (flat enum on the class itself)
# Qt6: QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow  (scoped enum)
if QT_MAJOR >= 6:
    FormFieldGrowthPolicy = QtWidgets.QFormLayout.FieldGrowthPolicy
else:
    FormFieldGrowthPolicy = QtWidgets.QFormLayout

# Qt.ItemIsEnabled flag
# Qt5: QtCore.Qt.ItemIsEnabled
# Qt6: QtCore.Qt.ItemFlag.ItemIsEnabled
if QT_MAJOR >= 6:
    ItemIsEnabled = QtCore.Qt.ItemFlag.ItemIsEnabled
else:
    ItemIsEnabled = QtCore.Qt.ItemIsEnabled

# QMessageBox standard buttons
# Qt5: QMessageBox.Yes / QMessageBox.Cancel
# Qt6: QMessageBox.StandardButton.Yes / QMessageBox.StandardButton.Cancel
MessageBoxStandardButton = getattr(
    QtWidgets.QMessageBox, "StandardButton", QtWidgets.QMessageBox
)
