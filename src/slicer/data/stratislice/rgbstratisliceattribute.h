#ifndef RGBStratiSliceAttribute_H
#define RGBStratiSliceAttribute_H

#include "abstractstratisliceattribute.h"
class IGraphicRepFactory;

class RGBStratiSliceAttribute: public AbstractStratiSliceAttribute {
Q_OBJECT
public:
	RGBStratiSliceAttribute(WorkingSetManager *workingSet,
			StratiSlice *stratislice, QObject *parent = 0);
	virtual ~RGBStratiSliceAttribute();

	void setExtractionWindow(uint w) override;

	int redIndex() const;
	int greenIndex() const;
	int blueIndex() const;

	void setRedIndex(int value);
	void setGreenIndex(int value);
	void setBlueIndex(int value);

	CUDARGBImage* image() {
		return m_image;
	}

	QString name() const override;

	//IData
	virtual IGraphicRepFactory* graphicRepFactory();
signals:
	void frequencyChanged();
protected:
	void loadSlice(unsigned int z) override;
	void resetFrequencies();
private:
	CUDARGBImage *m_image;
	unsigned int m_freqIndex[3];

	IGraphicRepFactory *m_repFactory;
};
Q_DECLARE_METATYPE(RGBStratiSliceAttribute*)
#endif
