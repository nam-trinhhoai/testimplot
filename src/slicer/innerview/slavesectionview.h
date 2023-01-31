#ifndef SlaveSectionView_H
#define SlaveSectionView_H

#include "abstractsectionview.h"

//Specialized graphic view to handle section: all the views need to be synchronized
class SlaveSectionView: public AbstractSectionView {
Q_OBJECT
public:
	SlaveSectionView(bool restictToMonoTypeSplit,ViewType type,
	QString uniqueName);
	virtual ~SlaveSectionView();
protected:
	void showRep(AbstractGraphicRep * rep) override;
	void hideRep(AbstractGraphicRep *rep) override;
	void cleanupRep(AbstractGraphicRep *rep) override;
};

#endif
