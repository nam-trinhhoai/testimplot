#ifndef FrequencyStratiSliceAttribute_H
#define FrequencyStratiSliceAttribute_H

#include <QObject>
#include "abstractstratisliceattribute.h"
class IGraphicRepFactory;
class FrequencyStratiSliceAttribute: public AbstractStratiSliceAttribute {
Q_OBJECT
public:
	FrequencyStratiSliceAttribute(WorkingSetManager *workingSet,
			StratiSlice *stratislice, QObject *parent = 0);
	virtual ~FrequencyStratiSliceAttribute();

	void setExtractionWindow(uint w) override;

	void setIndex(int value);
	int index() const ;

	int frequencyCount();


	CUDAImagePaletteHolder* image() {
		return m_currentImg;
	}

	QString name() const override;

	//IData
	virtual IGraphicRepFactory* graphicRepFactory();

signals:
	void indexChanged();
protected:
	void loadSlice(unsigned int z) override;
	void resetFrequencies();

	void updateExposedImage();
private:
	QVector<CUDAImagePaletteHolder*> m_images;
	IGraphicRepFactory *m_repFactory;

	CUDAImagePaletteHolder *m_currentImg;

	int m_currentIndex;
};
Q_DECLARE_METATYPE(FrequencyStratiSliceAttribute*)
#endif
