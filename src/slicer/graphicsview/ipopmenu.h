#ifndef IPOPMENU_H_
#define IPOPMENU_H_

#include <QMenu>
#include <QContextMenuEvent>

class IPopMenu {
public:
	virtual void fillContextMenu(QPointF scenePos, QContextMenuEvent::Reason, QMenu& mainMenu) = 0;
};

#endif
