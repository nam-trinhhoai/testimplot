#ifndef RGBLAYERFROMDATASET_H_
#define RGBLAYERFROMDATASET_H_

#include <QObject>
#include <QMutex>
#include <QString>

#include <memory>
#include "idata.h"
#include "cudaimagepaletteholder.h"

class IGraphicRepFactory;
class Seismic3DAbstractDataset;

class RgbLayerFromDataset :  public IData {
Q_OBJECT
public:
	RgbLayerFromDataset(QString name, WorkingSetManager *workingSet, Seismic3DAbstractDataset* dataset,
			QObject *parent = 0);
	virtual ~RgbLayerFromDataset();

	unsigned int width() const;
	unsigned int depth() const;
	unsigned int getNbProfiles() const;
	unsigned int getNbTraces() const;

	CUDAImagePaletteHolder* image(QString propName);
	bool writeProperty(float *tab, QString propName);
	bool readProperty(float *tab, QString propName);
	bool saveProperty(QString filename, QString propName);
	bool loadProperty(QString filename, QString propName);
	QVector<QString> keys();


	Seismic3DAbstractDataset* dataset() {
		return m_dataset;
	}

	float getStepSample();
	float getOriginSample();

	//IData
	virtual IGraphicRepFactory* graphicRepFactory() override;
	QUuid dataID() const override;
	QString name() const override;

	static const QString ISOCHRONE;

signals:
	void newPropertyCreated(QString);
	void propertyModified(QString);
private:
	std::unique_ptr<IGraphicRepFactory> m_repFactory;

	std::map<QString, std::shared_ptr<CUDAImagePaletteHolder>> m_images;
		Seismic3DAbstractDataset* m_dataset;
		QString m_name;
};

#endif
