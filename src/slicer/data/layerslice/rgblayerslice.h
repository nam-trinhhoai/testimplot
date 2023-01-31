#ifndef RGBLayerSlice_H
#define RGBLayerSlice_H

#include <QObject>
#include <QVector2D>
#include <memory>
#include "idata.h"
#include "isochronprovider.h"
#include "itreewidgetitemdecoratorprovider.h"

class IGraphicRepFactory;
class CUDARGBImage;
class LayerSlice;
class TextColorTreeWidgetItemDecorator;

class RGBLayerSlice: public IData, public IsoChronProvider, public ITreeWidgetItemDecoratorProvider {
	Q_OBJECT
public:
	RGBLayerSlice(WorkingSetManager *workingSet,
			LayerSlice *layerSlice, QObject *parent = 0);
	virtual ~RGBLayerSlice();


	int redIndex() const;
	int greenIndex() const;
	int blueIndex() const;

	void setRedIndex(int value);
	void setGreenIndex(int value);
	void setBlueIndex(int value);
	void setRGBIndexes(int red, int green, int blue);

	CUDARGBImage* image() {
		return m_image.get();
	}

	LayerSlice* layerSlice() {
		return m_layerSlice;
	}

	IsoSurfaceBuffer getIsoBuffer()override;

	bool isLocked() const;
	int lockedRedIndex() const;
	int lockedGreenIndex() const;
	int lockedBlueIndex() const;
	QVector2D lockedRedRange() const;
	QVector2D lockedGreenRange() const;
	QVector2D lockedBlueRange() const;

	void setLockedRedRange(const QVector2D& redRange);
	void setLockedGreenRange(const QVector2D& greenRange);
	void setLockedBlueRange(const QVector2D& blueRange);

	void unlock();
	void lock();

	// HSV minimum value and can be locked
	bool isMinimumValueActive() const;
	void setMinimumValueActive(bool active, bool bypassLock=false);
	float minimumValue() const;
	void setMinimumValue(float minimumValue, bool bypassLock=false);


	QUuid dataID() const override;
	QString name() const override;

	//IData
	virtual IGraphicRepFactory* graphicRepFactory() override;

	// ITreeWidgetItemDecoratorProvider
	virtual ITreeWidgetItemDecorator* getTreeWidgetItemDecorator() override;
	void resetFrequencies();
signals:
	void frequencyChanged();
	void deletedRep();
	void lockChanged();
	void minimumValueActivated(bool value);
	void minimumValueChanged(float value);

public slots:
	void deleteRep();
	void updateFromComputation();

protected:
	void loadSlice();
	int getMaximumEnergyComponent();
	int getMaximumEnergyComponent(float *tab, size_t size);
private:
	std::unique_ptr<CUDARGBImage> m_image;
	unsigned int m_freqIndex[3];

	LayerSlice* m_layerSlice;

	std::unique_ptr<IGraphicRepFactory> m_repFactory;

	bool m_locked = false;
	int m_lockedRedIndex;
	int m_lockedGreenIndex;
	int m_lockedBlueIndex;
	QVector2D m_lockedRedRange;
	QVector2D m_lockedGreenRange;
	QVector2D m_lockedBlueRange;

	bool m_useMinimumValue = false;
	float m_minimumValue = 0.0f; // from HSV standpoint

	TextColorTreeWidgetItemDecorator* m_decorator;
};
Q_DECLARE_METATYPE(RGBLayerSlice*)

#endif
