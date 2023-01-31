#ifndef WellBoreGraphicRepFactory_H
#define WellBoreGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class WellBore;

class WellBoreGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	WellBoreGraphicRepFactory(WellBore * data);
	virtual ~WellBoreGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;

private:
	WellBore * m_data;

};

#endif
