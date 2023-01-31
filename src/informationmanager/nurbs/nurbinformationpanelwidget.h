#ifndef SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONPANELWIDGET_H
#define SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONPANELWIDGET_H

#include "iinformationpanelwidget.h"

#include <QColor>
#include <QPointer>

class NurbInformation;

class QPushButton;

class NurbInformationPanelWidget : public IInformationPanelWidget {
	Q_OBJECT
public:
	NurbInformationPanelWidget(NurbInformation* information, QWidget* parent=nullptr);
	virtual ~NurbInformationPanelWidget();

	virtual bool saveChanges() override;

	QColor color() const;

public slots:
	void editColor();
	void setColor(QColor color);

private:
	void setButtonColor(const QColor& color);

	QPointer<NurbInformation> m_information;

	QColor m_color;
	QPushButton* m_colorButton;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONPANELWIDGET_H
