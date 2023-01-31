#ifndef HorizonFolderDataGraphicRepFactory_H
#define HorizonFolderDataGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class HorizonFolderData;
class IData;

class HorizonFolderDataGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	HorizonFolderDataGraphicRepFactory(HorizonFolderData* data);
	virtual ~HorizonFolderDataGraphicRepFactory();

	//Retreive the child of this data
//	virtual QList<IGraphicRepFactory *> childReps(ViewType type,AbstractInnerView * parent)  override;
	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;

//private slots:
	//void dataAdded(IData *data);
	//void dataRemoved(IData *data);
private:
	HorizonFolderData * m_data;

};

#endif
