#include "QtImGuiCore.h"

QtImGuiCore::QtImGuiCore(const QString& name, int time)
{
    this->name = name;
    this->time = time;
    QObject::connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
    timer.start(time);
}

QtImGuiCore::~QtImGuiCore()
{}

void QtImGuiCore::initializeGL()
{
    initializeOpenGLFunctions();
    ref = QtImGui::initialize(this, false);

    // Update at 60 fps
    auto* timer = new QTimer(this);
    QObject::connect(timer, SIGNAL(timeout()), this, SLOT(update()));
    timer->start(16);

    // create plot context
    ctx = ImPlot::CreateContext();
}


void QtImGuiCore::paintGL()
{
    QtImGui::newFrame(ref);
    ImPlot::SetCurrentContext(ctx);

	//ImPlot::ShowDemoWindow(&show_implot_demo_window);
    showPlot();

    // Do render before ImGui UI is rendered
    glViewport(0, 0, width(), height());
    glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui::Render();
    QtImGui::render(ref);
}

// Plot
void QtImGuiCore::showPlot()
{}

