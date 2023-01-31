#ifndef COLORTABLESELECTOR_H_
#define COLORTABLESELECTOR_H_

#include "colortablechooser.h"
#include "colortable.h"

#include <QPixmap>
#include <QWidget>

class ColorTableChooser;
class QLabel;
class QLineEdit;
class QPushButton;

class ColorTableSelector : public QWidget {
	Q_OBJECT
public:
	ColorTableSelector(QWidget* parent=0);
	virtual ~ColorTableSelector();

	void setSelection(const ColorTable&);
	// For ColorTableChooser only
	void changeSelection(const ColorTable& newSelection);
	const ColorTable& getCurrentSelection() const;

	static QPixmap getPixmap(const ColorTable& colorTable);

signals:
	void selectionChanged(ColorTable);

private slots:
    void openChooser();
    
private:
    QLabel* m_iconResult;
	QLineEdit* m_result;
	QPushButton* m_openDlg;
    ColorTableChooser* m_chooser;
    ColorTable m_curSelection;
};

#endif /* COLORTABLESELECTOR_H_ */
