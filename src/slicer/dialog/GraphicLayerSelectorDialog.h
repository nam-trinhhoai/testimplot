/*
 * GraphicLayerSelectorDialog.h
 *
 */

#ifndef TARUMAPP_SRC_DIALOG_GraphicLayerSelectorDialog_H_
#define TARUMAPP_SRC_DIALOG_GraphicLayerSelectorDialog_H_

#include <QDialog>
#include <QtGui>
#include <QStringList>

class QListWidget;

class QString;
//class QStringList;
class QListWidgetItem;
class Abstract2DInnerView;

class GraphicLayerSelectorDialog :public QDialog{
	Q_OBJECT
public:
	GraphicLayerSelectorDialog(QString, Abstract2DInnerView*);
	virtual ~GraphicLayerSelectorDialog();
	int getSelectedIndex() const;
	QString getSelectedString() const;

	private slots:
	void slotSelectItem( QListWidgetItem * current, QListWidgetItem * previous);
	void deleteFile();

	private:
	QStringList* m_stringList;
	QListWidget* m_listWidget;
	int m_selectedItem = -1;
	QString m_Path;
	Abstract2DInnerView* m_view;
};

#endif /* TARUMAPP_SRC_DIALOG_GraphicLayerSelectorDialog_H_ */
