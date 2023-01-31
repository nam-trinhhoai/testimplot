#ifndef WellHeadGraphicRepFactory_H
#define WellHeadGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class WellHead;
class WellBore;

class WellHeadGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	WellHeadGraphicRepFactory(WellHead * data);
	virtual ~WellHeadGraphicRepFactory();

	//Retreive the child of this data
	virtual QList<IGraphicRepFactory *> childReps(ViewType type,AbstractInnerView * parent)  override;
	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;

private slots:
	void wellBoreAdded(WellBore *wellBore);
	void wellBoreRemoved(WellBore *wellBore);
private:
	WellHead * m_data;

};

#endif
