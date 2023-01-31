#ifndef NurbsGraphicRepFactory_H
#define NurbsGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"
#include "nurbsdataset.h"

#include "igraphicrepfactory.h"
class NurbsDataset;


class NurbsGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	NurbsGraphicRepFactory(NurbsDataset * data);
	virtual ~NurbsGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;
	virtual QList<IGraphicRepFactory*> childReps(ViewType type,AbstractInnerView *parent) override;

//private slots:
//	void wellPickAdded(WellPick* wellPick);
	//void wellPickRemoved(WellPick* wellPick);

private:
	NurbsDataset * m_data;

};

#endif
