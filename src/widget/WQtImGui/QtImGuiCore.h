#ifndef __QT_IMGUI_CORE_H__
#define __QT_IMGUI_CORE_H__

#include <QtImGui.h>
#include <imgui.h>
#include <QGuiApplication>
#include <QTimer>
#include <QSurfaceFormat>
#include <QOpenGLWidget>
#include <QOpenGLExtraFunctions>

#include "implot.h"
#include "imgui.h"

class QtImGuiCore : public QOpenGLWidget
        , private QOpenGLExtraFunctions
{
public:
    QtImGuiCore() {};
    QtImGuiCore(const QString& name, int time);
    virtual ~QtImGuiCore();


    // Plot well logs
    virtual void showPlot();

protected:
    void initializeGL() override;
    void paintGL() override;

private:
    bool show_test_window = true;
    bool show_another_window = false;
    bool show_implot_demo_window = true;
    ImVec4 clear_color = ImColor(114, 144, 154);

    QString name;
    QtImGui::RenderRef ref = nullptr;
    int time;
    QTimer timer;
    ImPlotContext* ctx;
};

#endif
