#ifndef COLORTABLEDITDIALOG_H_
#define COLORTABLEDITDIALOG_H_

#include <QDialog>
#include "qhistogram.h"
#include "lookuptable.h"

class LUTWidget;
class QSlider;
class QAction;
class QActionGroup;
class QCheckBox;


class LUTEditDialog : public QDialog {
	Q_OBJECT
public:
	LUTEditDialog(const QHistogram & histo, const QVector2D & restrictedRange, const LookupTable & table,QWidget* parent=0);
	virtual ~LUTEditDialog();

	void setHistogramAndLookupTable(const QHistogram &histo, const QVector2D& restrictedRange,const LookupTable & table);
	void setLookupTable( const LookupTable & table );

signals:
	void lookupTableChanged(const LookupTable& colorTable);

private slots:
	void lookupTableFunctionParamsChanged(int p1, int p2);
	void lookupTableChangedInternal(const LookupTable& colorTable);
	void functionChanged(QAction *);
	void param1Changed(int );
	void param2Changed(int );

	void lookupPanelSizeChanged();


	void razTransp();
	void razFunction();
	void invert(int );

private:
	void createActions();
private:
	LUTWidget *m_lutEditor;
	QSlider *m_hSlider;
	QSlider *m_vSlider;

	QCheckBox * m_inverted;


	QAction *m_linearFctAction;
	QAction *m_binaryFctAction;
	QAction *m_binlinearFctAction;
	QAction *m_logFctAction;
	QAction *m_tri1Action;
	QAction *m_tri2Action;

	QActionGroup *m_functionGroup;
};



#endif /* QTLARGEIMAGEVIEWER_SRC_HISTOGRAM_COLORTABLEDITDIALOG_H_ */
