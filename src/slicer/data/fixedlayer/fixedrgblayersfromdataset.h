/*
 * FixedRGBLayersFromDataset.h
 *
 *  Created on: Oct. 02, 2020
 *      Author: l0483271
 */

#ifndef SRC_SLICER_DATA_FIXEDLAYER_FIXEDRGBLAYERSFROMDATASET_H_
#define SRC_SLICER_DATA_FIXEDLAYER_FIXEDRGBLAYERSFROMDATASET_H_

#include <QObject>
#include <QString>
#include <memory>

#include "idata.h"
#include "cudargbimage.h"

class IGraphicRepFactory;
class Seismic3DAbstractDataset;
class CUDAImagePaletteHolder;
class CUDARGBImage;

class FixedRGBLayersFromDataset: public IData {
Q_OBJECT
public:
	// cudaBuffer need to be a float RGBD planar stack
	FixedRGBLayersFromDataset(QList<std::pair<QString, QString>> layers, QString name,
			WorkingSetManager *workingSet, Seismic3DAbstractDataset* dataset,
			bool takeOwnership = true, QObject *parent = 0);
	virtual ~FixedRGBLayersFromDataset();

	unsigned int width() const;
	unsigned int depth() const;
	unsigned int getNbProfiles() const;
	unsigned int getNbTraces() const;


	Seismic3DAbstractDataset* dataset() {
		return m_dataset;
	}

	float getStepSample();
	float getOriginSample();

	// buffer access
	//std::size_t layerSize();
	const QList<std::vector<float>>& buffers() const;
	const QList<std::pair<QString, float*>>& layers() const;
	const QList<long>& selectedLayersKeys() const;
	void setSelectedLayersKeys(const QList<long>& newSelection);
	const QList<long>& layersKeys() const;
	void setLayersKeys(const QList<long>& newSelection);
	long currentImageIndex() const;
	void setCurrentImageIndex(long newIndex);

	//IData
	virtual IGraphicRepFactory* graphicRepFactory() override;
	QUuid dataID() const override;
	QString name() const override;


	CUDARGBImage* image() {
		return m_currentRGB.get();
	}

	CUDAImagePaletteHolder* isoSurfaceHolder() {
		return m_currentIso.get();
	}
	static FixedRGBLayersFromDataset* createDataFromDatasetWithUI(QString name,
			WorkingSetManager *workingSet, Seismic3DAbstractDataset* dataset,
			QObject *parent = 0);

signals:
	void layerSelectionChanged(const QList<long>& selectedLayers);
	void layerOrderChanged(const QList<long>& selectedLayers);

private:
	std::unique_ptr<IGraphicRepFactory> m_repFactory;

	Seismic3DAbstractDataset* m_dataset;
	QString m_name;

	QList<std::vector<float>> m_buffers;
	QList<std::pair<QString, float*>> m_layers;
	QList<long> m_layersKeys; // give order for layering
	QList<long> m_selectedLayersKeys; // must be an ordered subset of m_layersKeys

	std::unique_ptr<CUDAImagePaletteHolder> m_currentIso = nullptr;
	std::unique_ptr<CUDARGBImage> m_currentRGB = nullptr;
	long m_currentImageIndex = -1; // index for m_selectedLayersKeys
};

#endif /* SRC_SLICER_DATA_FIXEDLAYER_FIXEDLAYERFROMDATASET_H_ */
