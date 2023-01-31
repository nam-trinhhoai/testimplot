/*
 * undosystem.cpp
 *
 *  Created on: Sep 6, 2021
 *      Author: l1046262
 */

#include "GraphicSceneEditor.h"
#include "undosystem.h"
#include <QDebug>

void UndoSystem::backup(const QList<QGraphicsItem*>&& items)
{
    int stackSize = itemsStack.length();
    if (currentIndex < stackSize - 1)
    {
        for (int i = currentIndex + 1; i < stackSize; ++i)
        {
            free(itemsStack[i]);
        }
        itemsStack.erase(itemsStack.begin() + currentIndex + 1, itemsStack.end());
    }

    itemsStack.push_back(items);
    currentIndex++;
}

QList<QGraphicsItem*> UndoSystem::undo()
{
    return itemsStack[--currentIndex];
}

QList<QGraphicsItem*> UndoSystem::redo()
{
    return itemsStack[++currentIndex];
}

void UndoSystem::free(QList<QGraphicsItem*> const& items)
{
    foreach(QGraphicsItem* p, items) {
        delete p;
    }
}
