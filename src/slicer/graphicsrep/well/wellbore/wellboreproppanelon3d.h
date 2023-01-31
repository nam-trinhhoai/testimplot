#ifndef WellBorePropPanelOn3D_H
#define WellBorePropPanelOn3D_H

#include <QWidget>

class WellBoreRepOn3D;
class QComboBox;
class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLabel;

class WellBorePropPanelOn3D : public QWidget {
	Q_OBJECT
public:
	WellBorePropPanelOn3D(WellBoreRepOn3D* rep, QWidget* parent=0);
	WellBorePropPanelOn3D(WellBoreRepOn3D* rep, long defaultWidth, long minimalWidth, long maximalWidth,
						double logMin, double logMax, QColor defaultColor, QWidget* parent=0);
	~WellBorePropPanelOn3D();

	long defaultWidth() const;
	long minimalWidth() const;
	long maximalWidth() const;
	double logMin() const;
	double logMax() const;
	QColor defaultColor() const;


signals:
	/*void logChanged(long index);
	void defaultWidthChanged(int defaultWidth);
	void minimalWidthChanged(int minimalWidth);
	void maximalWidthChanged(int maximalWidth);
	void logMinChanged(double mini);
	void logMaxChanged(double maxi);
	void defaultColorChanged(QColor color);*/
	void stateUpdated();

public slots:
	void updateLog();
	void updateWidget();
	void setLog(long index);
	void setDefaultWidth(int defaultWidth);
	void setMinimalWidth(int minimalWidth);
	void setMaximalWidth(int maximalWidth);
	void setLogMin(double mini);
	void setLogMax(double maxi);
	void setDefaultColor(QColor color);

private slots:
	void logChangedInternal(int index);
	void defaultWidthChangedInternal(int defaultWidth);
	void minimalWidthChangedInternal(int minimalWidth);
	void maximalWidthChangedInternal(int maximalWidth);
	void logMinChangedInternal(double mini);
	void logMaxChangedInternal(double maxi);
	void defaultColorChangedInternal();
	void applyChanges();
	void restoreState();
	void checkCurrentState();

private:
	void buildWidget(); // only call once in constructor
	void updateState();

	void computeMinMax();

	WellBoreRepOn3D* m_rep;
	QComboBox* m_logComboBox;
	QSpinBox* m_defaultWidthSpinBox;
	QSpinBox* m_minimalWidthSpinBox;
	QSpinBox* m_maximalWidthSpinBox;
	QDoubleSpinBox* m_logMinSpinBox;
	QDoubleSpinBox* m_logMaxSpinBox;
	QPushButton* m_defaultColorHolder;
	QLabel* m_defaultColorLabel;

	QPushButton* m_applyChanges;
	QPushButton* m_restoreState;

	long m_noLogIndex;
	long m_logIndex;
	long m_defaultWidth;
	long m_minimalWidth;
	long m_maximalWidth;
	double m_logMin;
	double m_logMax;
	QColor m_defaultColor;

	long m_logIndexSave;
	long m_defaultWidthSave;
	long m_minimalWidthSave;
	long m_maximalWidthSave;
	double m_logMinSave;
	double m_logMaxSave;
	QColor m_defaultColorSave;
};

#endif
