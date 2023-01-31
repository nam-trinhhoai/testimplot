#ifndef SeismicSurvey_H
#define SeismicSurvey_H
#include <QObject>
#include <QVector2D>

#include "idata.h"
#include "igeorefimage.h"
#include "ifilebaseddata.h"

#include "affine2dtransformation.h"

class Seismic3DAbstractDataset;
class SeismicSurveyGraphicRepFactory;
class SeismicSurvey: public IData, public IFileBasedData {
Q_OBJECT
public:
	SeismicSurvey(WorkingSetManager * workingSet,const QString &name, int width, int height, QString idPath, QObject *parent =
			0);
	virtual ~SeismicSurvey();

	const Affine2DTransformation  * const inlineXlineToXYTransfo() const;
	const Affine2DTransformation  * const ijToXYTransfo() const;

	void setInlineXlineToXYTransfo( const Affine2DTransformation& transfo);
	void setIJToXYTransfo( const Affine2DTransformation& transfo);

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}

	void addDataset(Seismic3DAbstractDataset *dataset);
	void removeDataset(Seismic3DAbstractDataset *dataset);
	QList<Seismic3DAbstractDataset*> datasets();
signals:
	void datasetAdded(Seismic3DAbstractDataset *dataset);
	void datasetRemoved(Seismic3DAbstractDataset *dataset);
private:
	unsigned int m_width;
	unsigned int m_height;

	QString m_name;
	QUuid m_uuid;

	Affine2DTransformation *m_ilXlToXY;
	Affine2DTransformation *m_ijToXY;

	QList<Seismic3DAbstractDataset*> m_datasets;
	SeismicSurveyGraphicRepFactory * m_repFactory;
};

#endif /* QTCUDAIMAGEVIEWER_QGLTEXTURE_H_ */
