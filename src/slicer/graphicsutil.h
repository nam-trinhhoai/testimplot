#ifndef GraphicsUtil_H
#define GraphicsUtil_H
#include <QString>
class QPushButton;
class QWidget;

class GraphicsUtil {
public:
	~GraphicsUtil() {
	}

	static QPushButton* generateToobarButton(const QString &iconPath,
			const QString &tooltip, QWidget *parent);

	static void arrangeWindows();
private:
	GraphicsUtil() {
	}
};

#endif
