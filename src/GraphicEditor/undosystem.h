/*
 * undosystem.h
 *
 *  Created on: Sep 6, 2021
 *      Author: l1046262
 */

#ifndef SRC_GENERICEDITOR_GRAPHEDITOR_UNDOSYSTEM_H_
#define SRC_GENERICEDITOR_GRAPHEDITOR_UNDOSYSTEM_H_

#include <QGraphicsItem>
#include <QList>

class UndoSystem {
public:
    void backup(QList<QGraphicsItem*> const&& items);
    QList<QGraphicsItem*> undo();
    QList<QGraphicsItem*> redo();
    bool isEmpty() {return currentIndex < 1;}
    bool isFull() {return currentIndex + 1 == itemsStack.length();}

private:
    void free(QList<QGraphicsItem*> const& items);
    QList<QList<QGraphicsItem*>> itemsStack;
    int currentIndex = -1;
};
Q_DECLARE_METATYPE( UndoSystem )
#endif /* SRC_GENERICEDITOR_GRAPHEDITOR_UNDOSYSTEM_H_ */
