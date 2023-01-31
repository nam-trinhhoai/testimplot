#ifndef AbstractStratiSliceAttribute_H
#define AbstractStratiSliceAttribute_H

/**
 * Modification code to adapt to dataset sampleType not done
 * Modification code to adapt to dataset channel not done
 *
 * It was not done because, I do not know this code, code that must contain a lot of cuda code.
 * I think it will take too long and the command was to avoid too long modifications
 *
 * Armand Sibille L0483271 19/02/2021
 */

#include <QObject>
#include "idata.h"

class Seismic3DAbstractDataset;
class CUDAImagePaletteHolder;
class CUDARGBImage;
class StratiSlice;

class AbstractStratiSliceAttribute: public IData {
Q_OBJECT
public:
	AbstractStratiSliceAttribute(WorkingSetManager *workingSet,
			StratiSlice *slice, QObject *parent = 0);
	virtual ~AbstractStratiSliceAttribute();

	int currentPosition() const;
	void setSlicePosition(int pos);

	uint extractionWindow() const;
	virtual void setExtractionWindow(uint w);

	//Hacky: calculation is expensive. By default, nothing is loaded within the objects.
	//Calling this method force to load (if needed...) a first layer.
	//Typically before the intial display
	void initialize();

	CUDAImagePaletteHolder* isoSurfaceHolder();

	StratiSlice* stratiSlice() const {
		return m_stratiSlice;
	}

	//IData
	QUuid dataID() const override;
	QString name() const override;
signals:
	void extractionWindowChanged(unsigned int size);
	void RGTIsoValueChanged(int pos);
protected:
	virtual void loadSlice(unsigned int z);
	void loadSlice(CUDAImagePaletteHolder *isoSurfaceImage,
			CUDAImagePaletteHolder *image, unsigned int extractionWindow,
			unsigned int z);

	void loadConcurrentSlice(CUDAImagePaletteHolder *isoSurfaceImage,
			CUDAImagePaletteHolder *image, unsigned int extractionWindow,
			unsigned int z);

	void loadRGBSlice(CUDAImagePaletteHolder *isoSurfaceImage,
			CUDARGBImage *image, unsigned int extractionWindow, unsigned int z,
			int f1, int f2, int f3);
	void loadConcurrentRGBSlice(CUDAImagePaletteHolder *isoSurfaceImage,
			CUDARGBImage *image, unsigned int extractionWindow, unsigned int z,
			int f1, int f2, int f3);


	void loadFrequencySlice(CUDAImagePaletteHolder *isoSurfaceImage,
			QVector<CUDAImagePaletteHolder *>images, unsigned int extractionWindow, unsigned int z);
	void loadConcurrentFrequencySlice(CUDAImagePaletteHolder *isoSurfaceImage,
			QVector<CUDAImagePaletteHolder *>images, unsigned int extractionWindow, unsigned int z);

protected:
	unsigned int m_extractionWindow;
	unsigned int m_currentSlice;

	StratiSlice *m_stratiSlice;

	CUDAImagePaletteHolder *m_isoSurfaceHolder;

	bool m_isInitialized;

};
Q_DECLARE_METATYPE(AbstractStratiSliceAttribute*)
#endif
