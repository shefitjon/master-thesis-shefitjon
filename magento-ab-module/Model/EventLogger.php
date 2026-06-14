<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Model;

class EventLogger
{
    private const TABLE = 'bregu_cart_recovery_event';

    public function __construct(
        private readonly \Magento\Framework\App\ResourceConnection $resource
    ) {
    }

    public function recordExposure(string $sessionKey, string $bucket, float $risk, string $signal): void
    {
        $connection = $this->resource->getConnection();
        $connection->insertOnDuplicate(
            $this->resource->getTableName(self::TABLE),
            [
                'session_key' => $sessionKey,
                'bucket' => $bucket,
                'risk_score' => $risk,
                'signal' => $signal,
            ],
            ['bucket', 'risk_score', 'signal']
        );
    }

    public function markShown(string $sessionKey): void
    {
        $this->update($sessionKey, ['shown' => 1]);
    }

    public function markDismissed(string $sessionKey): void
    {
        $this->update($sessionKey, ['dismissed' => 1]);
    }

    public function recordConversion(string $sessionKey, int $orderId, float $orderValue): void
    {
        $this->update($sessionKey, [
            'converted' => 1,
            'order_id' => $orderId,
            'order_value' => $orderValue,
        ]);
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    public function aggregateByBucket(): array
    {
        $connection = $this->resource->getConnection();
        $select = $connection->select()
            ->from(
                $this->resource->getTableName(self::TABLE),
                [
                    'bucket',
                    'exposures' => new \Magento\Framework\DB\Sql\Expression('COUNT(*)'),
                    'impressions' => new \Magento\Framework\DB\Sql\Expression('SUM(shown)'),
                    'conversions' => new \Magento\Framework\DB\Sql\Expression('SUM(converted)'),
                    'revenue' => new \Magento\Framework\DB\Sql\Expression('SUM(order_value)'),
                ]
            )
            ->group('bucket');

        return $connection->fetchAll($select);
    }

    /**
     * @param array<string, mixed> $data
     */
    private function update(string $sessionKey, array $data): void
    {
        $connection = $this->resource->getConnection();
        $connection->update(
            $this->resource->getTableName(self::TABLE),
            $data,
            ['session_key = ?' => $sessionKey]
        );
    }
}
