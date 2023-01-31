/*
 * SelectionRectangle.h
 *
 *  Created on: 16 févr. 2018
 *      Author: j0483271
 */

#ifndef MURATAPP_SRC_VIEW_CANVAS2D_SELECTIONRECTANGLE_H_
#define MURATAPP_SRC_VIEW_CANVAS2D_SELECTIONRECTANGLE_H_

#include "rectanglemovable.h"
#include "Node.h"

#include <vector>
#include <memory>

namespace murat {
namespace gui {

class SelectionRectangle: public RectangleMovable {
    // Use signals and slots
    Q_OBJECT
	Q_INTERFACES(QGraphicsItem)
public:
    SelectionRectangle(qreal rx=1, qreal ry=1, QObject* parent=0);
	virtual ~SelectionRectangle();

public slots:
	void nodeModification(Node* node, QPointF newPos, QPointF oldPos);

public:// to put back to protected!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	void updateNodes();

	// tl is 0
	// tr is 1
	// br is 2
	// bl is 3
	Node* nodes[4];

	bool lock = false;
};

} /* namespace gui */
} /* namespace murat */

#endif /* MURATAPP_SRC_VIEW_CANVAS2D_SELECTIONRECTANGLE_H_ */
