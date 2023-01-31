#ifndef SRC_INFORMATIONMANAGER_SEISMIC_SEISMICINFORMATIONPANELWIDGET_H
#define SRC_INFORMATIONMANAGER_SEISMIC_SEISMICINFORMATIONPANELWIDGET_H

#include "iinformationpanelwidget.h"

#include <QColor>
#include <QPointer>
#include <QWidget>

class SeismicInformation;
class QHBoxLayout;

class QPushButton;

class SeismicInformationPanelWidget : public IInformationPanelWidget {
	Q_OBJECT
public:
	SeismicInformationPanelWidget(SeismicInformation* information, QWidget* parent=nullptr);
	virtual ~SeismicInformationPanelWidget();

	virtual bool saveChanges() override;

	QColor color() const;

public slots:
	void editColor();
	void setColor(QColor color);

private:
	void setButtonColor(const QColor& color);

	QPointer<SeismicInformation> m_information;

	QColor m_color;
	// QPushButton* m_colorButton;
	QHBoxLayout *infoDimensionCreate(float vmin, float vmax, float step, QString format, QString suffix);
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONPANELWIDGET_H
