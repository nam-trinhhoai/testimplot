
#ifndef __FREEHORIZONATTRIBUTREPFACTORY__
#define __FREEHORIZONATTRIBUTREPFACTORY__

#include <QObject>
#include "viewutils.h"

#include <fixedrgblayersfromdatasetandcube.h>
#include "igraphicrepfactory.h"

class FreeHorizonAttribut;
class FreeHorizonAttributLayer;


class FreeHorizonAttributRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	FreeHorizonAttributRepFactory(FreeHorizonAttribut * data);
	virtual ~FreeHorizonAttributRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type, AbstractInnerView * parent)  override;

private slots:
		void freeHorizonAttributAdded();

private:
	FreeHorizonAttribut * m_data = nullptr;
	FreeHorizonAttributLayer *m_layer = nullptr;

};

#endif
