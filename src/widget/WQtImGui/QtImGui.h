#ifndef __QT_IMGUI_H__
#define __QT_IMGUI_H__

class QWidget;
class QWindow;

namespace QtImGui {

typedef void* RenderRef;

RenderRef initialize(QWidget *window, bool defaultRender = true);

RenderRef initialize(QWindow *window, bool defaultRender = true);
void newFrame(RenderRef ref = nullptr);
void render(RenderRef ref = nullptr);

}

#endif
