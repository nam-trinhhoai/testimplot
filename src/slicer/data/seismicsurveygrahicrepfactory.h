#ifndef SeismicSurveyGraphicRepFactory_H
#define SeismicSurveyGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class SeismicSurvey;
class Seismic3DAbstractDataset;

class SeismicSurveyGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	SeismicSurveyGraphicRepFactory(SeismicSurvey * data);
	virtual ~SeismicSurveyGraphicRepFactory();

	//Retreive the child of this data
	virtual QList<IGraphicRepFactory *> childReps(ViewType type,AbstractInnerView * parent)  override;
	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;

private slots:
	void datasetAdded(Seismic3DAbstractDataset *dataset);
	void datasetRemoved(Seismic3DAbstractDataset *dataset);
private:
	SeismicSurvey * m_data;

};

#endif
