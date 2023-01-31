#ifndef AmplitudeStratiSliceAttribute_H
#define AmplitudeStratiSliceAttribute_H

#include "abstractstratisliceattribute.h"
class IGraphicRepFactory;

class AmplitudeStratiSliceAttribute: public AbstractStratiSliceAttribute {
Q_OBJECT
public:
	AmplitudeStratiSliceAttribute(WorkingSetManager *workingSet, StratiSlice *slice,
			QObject *parent = 0);
	virtual ~AmplitudeStratiSliceAttribute();

	CUDAImagePaletteHolder* image() {
		return m_image;
	}
	QString name() const override;

	//IData
	virtual IGraphicRepFactory* graphicRepFactory();

protected:
	void loadSlice(unsigned int z) override;
private:
	CUDAImagePaletteHolder *m_image;

	IGraphicRepFactory *m_repFactory;
};
Q_DECLARE_METATYPE(AmplitudeStratiSliceAttribute*)
#endif
