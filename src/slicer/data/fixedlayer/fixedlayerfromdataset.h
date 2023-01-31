/*
 * LayerSlice.h
 *
 *  Created on: Jun 15, 2020
 *      Author: l0222891
 */

#ifndef SRC_SLICER_DATA_FIXEDLAYER_FIXEDLAYERFROMDATASET_H_
#define SRC_SLICER_DATA_FIXEDLAYER_FIXEDLAYERFROMDATASET_H_

#include <QObject>
#include <QMutex>
#include <QString>

#include <memory>
#include "idata.h"
#include "iGraphicToolDataControl.h"
#include "cudaimagepaletteholder.h"
#include "isochronprovider.h"
#include "itreewidgetitemdecoratorprovider.h"
class IGraphicRepFactory;
class Seismic3DAbstractDataset;
class TextColorTreeWidgetItemDecorator;


class FixedLayerFromDataset: public IData, public iGraphicToolDataControl, public IsoChronProvider,
		public ITreeWidgetItemDecoratorProvider {
Q_OBJECT
public:
	FixedLayerFromDataset(QString name, WorkingSetManager *workingSet, Seismic3DAbstractDataset* dataset,
			QObject *parent = 0);
	virtual ~FixedLayerFromDataset();

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

	IsoSurfaceBuffer getIsoBuffer()override;

	//IData
	virtual IGraphicRepFactory* graphicRepFactory() override;
	QUuid dataID() const override;
	QString name() const override;

	// iGraphicToolDataControl
	// expect item to be in basemap coordinate system
	virtual void deleteGraphicItemDataContent(QGraphicsItem* item) override;

	// ITreeWidgetItemDecoratorProvider
	virtual ITreeWidgetItemDecorator* getTreeWidgetItemDecorator() override;

	// color management
	QColor getColor() const;
	void setColor(const QColor& color);
	bool loadColor(const QString& colorFilePath);
	bool saveColor(const QString& colorFilePath) const;

	// This class can manage temporary data and data from files.
	// To keep the m_decorator up to date, the object managing the FixedLayerFromDataset object
	// has to update "toggleTemporaryData"
	bool isTemporaryData() const;
	void toggleTemporaryData(bool val);

	static QColor loadColorFromFile(const QString& colorFilePath, bool* ok=nullptr);
	static bool saveColorToFile(const QString& colorFilePath, const QColor& color);


	static const QString ISOCHRONE;

protected:
	std::unique_ptr<IGraphicRepFactory> m_repFactory;
	QString m_name;
	Seismic3DAbstractDataset* m_dataset;
	QColor m_color;



signals:
	void newPropertyCreated(QString);
	void propertyModified(QString);
	void colorChanged(QColor color);
	void isTemporaryDataChanged(bool val);
private:

	std::map<QString, std::shared_ptr<CUDAImagePaletteHolder>> m_images;

	TextColorTreeWidgetItemDecorator* m_decorator;
	bool m_isTemporaryData;
};

#endif /* SRC_SLICER_DATA_FIXEDLAYER_FIXEDLAYERFROMDATASET_H_ */
