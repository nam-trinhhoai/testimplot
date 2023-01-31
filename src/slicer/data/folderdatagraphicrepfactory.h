#ifndef FolderDataGraphicRepFactory_H
#define FolderDataGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class FolderData;
class IData;

class FolderDataGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	FolderDataGraphicRepFactory(FolderData* data);
	virtual ~FolderDataGraphicRepFactory();

	//Retreive the child of this data
	virtual QList<IGraphicRepFactory *> childReps(ViewType type,AbstractInnerView * parent)  override;
	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;

private slots:
	void dataAdded(IData *data);
	void dataRemoved(IData *data);
private:
	FolderData * m_data;

};

#endif
