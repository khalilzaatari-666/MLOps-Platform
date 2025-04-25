ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY 'root';
ALTER USER 'khalil'@'%' IDENTIFIED WITH mysql_native_password BY 'khalil123';
CREATE DATABASE IF NOT EXISTS `pcsagri` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;