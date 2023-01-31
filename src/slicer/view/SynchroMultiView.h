/*
 * SynchroMultiView.h
 *
 *  Created on: Nov 23, 2020
 *      Author: l0222891
 */

#ifndef SRC_SLICER_VIEW_SYNCHROMULTIVIEW_H_
#define SRC_SLICER_VIEW_SYNCHROMULTIVIEW_H_

#include <QObject>
#include <QVector>

class AbstractInnerView;

class SynchroMultiView : public QObject {
Q_OBJECT

public:
	SynchroMultiView();
	virtual ~SynchroMultiView();

	void registerView(AbstractInnerView* abstractInnerView);
	void unregisterView(AbstractInnerView* abstractInnerView);
	void removeView(AbstractInnerView* abstractInnerView);

	bool isSameSlice() const {
		return (m_synchroType == 1);
	}

	void setSynchroType(int synchroType) {
		m_synchroType = synchroType;
	}

public slots:
	void sliceChangedInOtherView(int value, int delta,
			AbstractInnerView* originView);

private:
	QVector<AbstractInnerView *> m_viewVect;
	int m_synchroType = 1;
};

#endif /* SRC_SLICER_VIEW_SYNCHROMULTIVIEW_H_ */
