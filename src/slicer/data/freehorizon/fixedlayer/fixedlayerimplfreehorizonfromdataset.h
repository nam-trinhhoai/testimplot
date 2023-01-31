

#ifndef __FIXEDLAYERIMPLFREEHORIZONFROMDATASET__
#define __FIXEDLAYERIMPLFREEHORIZONFROMDATASET__


#include <QObject>
#include <QMutex>
#include <QString>

#include <memory>
#include "idata.h"
#include "iGraphicToolDataControl.h"
#include "cudaimagepaletteholder.h"
#include "isochronprovider.h"

#include "fixedlayerfromdataset.h"

class IGraphicRepFactory;


class FixedLayerImplFreeHorizonFromDataset : public FixedLayerFromDataset {
Q_OBJECT
public:
	FixedLayerImplFreeHorizonFromDataset(QString name, WorkingSetManager *workingSet, Seismic3DAbstractDataset* dataset, QObject *parent = 0);
	virtual ~FixedLayerImplFreeHorizonFromDataset();


	IGraphicRepFactory* graphicRepFactory();
	void deleteGraphicItemDataContent(QGraphicsItem* item);


	QString name();




};




#endif
