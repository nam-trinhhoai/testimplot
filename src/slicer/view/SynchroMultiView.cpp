/*
 * SynchroMultiView.cpp
 *
 *  Created on: Nov 23, 2020
 *      Author: l0222891
 */

#include "SynchroMultiView.h"

#include <iostream>

#include "abstractinnerview.h"
#include "singlesectionview.h"
#include "slavesectionview.h"

SynchroMultiView::SynchroMultiView() {


}

SynchroMultiView::~SynchroMultiView() {

}

void SynchroMultiView::registerView(AbstractInnerView* abstractInnerView) {
	SingleSectionView* inlineAbstractView = dynamic_cast<SingleSectionView*>(abstractInnerView);
	if ( ! inlineAbstractView)
		return;
	if (m_viewVect.contains(abstractInnerView))
		return;

	m_viewVect.append(abstractInnerView);

	ViewType viewType = inlineAbstractView->viewType();
	//if (inlineAbstractView->getViewType() == ViewType::InlineView)
	/*connect(abstractInnerView, SIGNAL(sliceChangedFromView(int value, int delta,
			AbstractInnerView* originView)), this,
			SLOT(sliceChangedInOtherView(int value, int delta,
					AbstractInnerView* originView)));*/
	connect(abstractInnerView, &SingleSectionView::sliceChangedFromView, this,
			&SynchroMultiView::sliceChangedInOtherView);
}

void SynchroMultiView::sliceChangedInOtherView(int value, int delta, AbstractInnerView* originView) {
	for (AbstractInnerView* view : m_viewVect) {
		if ( view == originView)
			continue; // View at the origin
		if ( view->viewType() != originView->viewType())
			continue; // Type is diffrent (example inline versus xline)
		SingleSectionView* sectionView = dynamic_cast<SingleSectionView*>(view);

		if (sectionView) {
			if (this->m_synchroType==1)
				sectionView->sliceChangedFromOther(value, false );
			else if (this->m_synchroType==2)
				sectionView->sliceChangedFromOther(delta, true );
		}
	}
}

void SynchroMultiView::unregisterView(AbstractInnerView* abstractInnerView) {
	SingleSectionView* inlineAbstractView = dynamic_cast<SingleSectionView*>(abstractInnerView);
	if ( ! inlineAbstractView)
		return;
	if (m_viewVect.contains(abstractInnerView)) {
		/*disconnect(abstractInnerView, SIGNAL(SingleSectionView::sliceChangedFromView(int value, int delta,
				AbstractInnerView* originView)), this,
				SLOT(sliceChangedInOtherView(int value, int delta,
						AbstractInnerView* originView)));*/
		disconnect(abstractInnerView, &SingleSectionView::sliceChangedFromView, this,
					&SynchroMultiView::sliceChangedInOtherView);
		m_viewVect.removeOne(abstractInnerView);
	}
}


void SynchroMultiView::removeView(AbstractInnerView* abstractInnerView) {

}

