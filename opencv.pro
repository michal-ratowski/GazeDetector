TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
QT += gui

SOURCES += main.cpp
LIBS += `pkg-config opencv --libs`

