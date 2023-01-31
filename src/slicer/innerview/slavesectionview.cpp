#include "slavesectionview.h"
#include "slicerep.h"
#include "isliceablerep.h"
#include "cudaimagepaletteholder.h"

SlaveSectionView::SlaveSectionView(bool restictToMonoTypeSplit,ViewType type, 
QString uniqueName) :
AbstractSectionView(restictToMonoTypeSplit,type, uniqueName){
}


void SlaveSectionView::showRep(AbstractGraphicRep *rep) {
	//We need to add axis
	if (SliceRep *slice = dynamic_cast<SliceRep*>(rep)) {
		if (firstSlice() == nullptr) {
			addAxis(slice->image());
			defineScale(slice);

			//updateTile(slice->name());
			m_currentSliceIJPosition=slice->currentSliceIJPosition();
			m_currentSliceWorldPosition=slice->currentSliceWorldPosition();
		}
	}
	AbstractSectionView::showRep(rep);
	updateVerticalAxisColor();
	updateTitleFromSlices();
}

void SlaveSectionView::hideRep(AbstractGraphicRep *rep) {
	AbstractSectionView::hideRep(rep);
	if (firstSlice() == nullptr) {
		removeAxis();
	}
	updateTitleFromSlices();
}

void SlaveSectionView::cleanupRep(AbstractGraphicRep *rep) {
	AbstractSectionView::cleanupRep(rep);
	if (firstSlice() == nullptr) {
		removeAxis();
	}
	updateTitleFromSlices();
}
SlaveSectionView::~SlaveSectionView() {

}
