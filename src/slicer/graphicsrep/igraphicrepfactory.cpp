#include "igraphicrepfactory.h"

IGraphicRepFactory::IGraphicRepFactory(QObject *parent) :
		QObject(parent) {

}

IGraphicRepFactory::~IGraphicRepFactory() {

}

QList<IGraphicRepFactory*> IGraphicRepFactory::childReps(ViewType type,
		AbstractInnerView *parent) {
	return QList<IGraphicRepFactory*>();
}
