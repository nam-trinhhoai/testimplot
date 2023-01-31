#ifndef WellPickGraphicRepFactory_H
#define WellPickGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class WellPick;

class WellPickGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	WellPickGraphicRepFactory(WellPick * data);
	virtual ~WellPickGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;

private:
	WellPick * m_data;

};

#endif
