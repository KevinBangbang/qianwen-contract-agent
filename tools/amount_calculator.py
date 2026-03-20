"""
金额和日期计算工具

处理合同中涉及的数值计算，包括：
- 违约金计算（按比例、按日计算等）
- 付款周期和到期日计算
- 金额大小写转换和验证

这是唯一不依赖LLM的工具，纯Python逻辑实现。
这样做的好处是：数值计算交给代码比交给LLM更准确，LLM容易算错数。
"""

import json
from datetime import datetime, timedelta

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('amount_calculator')
class AmountCalculator(BaseTool):
    description = '合同金额和日期计算工具，支持违约金计算、付款周期计算、金额验证等纯数值运算'
    parameters = [{
        'name': 'calculation_type',
        'type': 'string',
        'description': '计算类型，可选值: penalty（违约金）, payment_schedule（付款周期）, date_diff（日期间隔）, amount_verify（金额验证）',
        'required': True,
    }, {
        'name': 'params',
        'type': 'object',
        'description': '计算参数，不同计算类型需要不同的参数',
        'required': True,
    }]

    def call(self, params: str, **kwargs) -> str:
        """
        根据计算类型执行对应的数值计算

        将复杂的数值逻辑从LLM中分离出来，用确定性的代码实现，
        避免LLM在数学计算上的不稳定性。
        """
        params = self._verify_json_format_args(params)
        calc_type = params.get('calculation_type', '')
        calc_params = params.get('params', {})

        # 根据计算类型路由到对应的方法
        handlers = {
            'penalty': self._calculate_penalty,
            'payment_schedule': self._calculate_payment_schedule,
            'date_diff': self._calculate_date_diff,
            'amount_verify': self._verify_amount,
        }

        handler = handlers.get(calc_type)
        if not handler:
            return json.dumps({
                'success': False,
                'error': f'不支持的计算类型: {calc_type}，'
                         f'支持的类型: {", ".join(handlers.keys())}'
            }, ensure_ascii=False)

        try:
            result = handler(calc_params)
            return json.dumps({
                'success': True,
                'calculation_type': calc_type,
                'result': result
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': f'计算失败: {str(e)}'
            }, ensure_ascii=False)

    def _calculate_penalty(self, params: dict) -> dict:
        """
        计算违约金

        支持两种常见的违约金计算方式：
        1. 固定比例：违约金 = 合同金额 * 违约比例
        2. 按日计算：违约金 = 合同金额 * 日利率 * 逾期天数
        """
        amount = float(params.get('contract_amount', 0))
        penalty_rate = float(params.get('penalty_rate', 0))
        overdue_days = int(params.get('overdue_days', 0))
        # daily表示按日计算，fixed表示固定比例
        calc_method = params.get('method', 'fixed')

        if calc_method == 'daily':
            # 按日计算：日利率 * 天数
            daily_rate = penalty_rate / 365
            penalty = amount * daily_rate * overdue_days
            return {
                'penalty_amount': round(penalty, 2),
                'daily_rate': round(daily_rate, 6),
                'overdue_days': overdue_days,
                'formula': f'{amount} * {daily_rate:.6f} * {overdue_days}',
                # 违约金超过本金30%的提醒（参照司法实践）
                'warning': '违约金超过合同金额的30%，可能被认定为过高'
                if penalty > amount * 0.3 else None
            }
        else:
            # 固定比例
            penalty = amount * penalty_rate
            return {
                'penalty_amount': round(penalty, 2),
                'penalty_rate': penalty_rate,
                'formula': f'{amount} * {penalty_rate}',
                'warning': '违约金超过合同金额的30%，可能被认定为过高'
                if penalty > amount * 0.3 else None
            }

    def _calculate_payment_schedule(self, params: dict) -> dict:
        """
        生成付款计划表

        根据总金额、期数和起始日期，生成每期的付款时间和金额。
        """
        total_amount = float(params.get('total_amount', 0))
        installments = int(params.get('installments', 1))
        start_date_str = params.get('start_date', '')

        # 解析起始日期
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            start_date = datetime.now()

        # 计算每期金额（均分，最后一期补差额）
        per_amount = round(total_amount / installments, 2)
        schedule = []
        for i in range(installments):
            # 每期间隔30天
            pay_date = start_date + timedelta(days=30 * (i + 1))
            # 最后一期补差额，确保总额精确
            if i == installments - 1:
                current_amount = round(
                    total_amount - per_amount * (installments - 1), 2
                )
            else:
                current_amount = per_amount

            schedule.append({
                'installment': i + 1,
                'due_date': pay_date.strftime('%Y-%m-%d'),
                'amount': current_amount,
            })

        return {
            'total_amount': total_amount,
            'installments': installments,
            'per_amount': per_amount,
            'schedule': schedule,
        }

    def _calculate_date_diff(self, params: dict) -> dict:
        """
        计算两个日期之间的间隔

        用于验证合同期限、计算逾期天数等场景。
        """
        start_str = params.get('start_date', '')
        end_str = params.get('end_date', '')

        start = datetime.strptime(start_str, '%Y-%m-%d')
        end = datetime.strptime(end_str, '%Y-%m-%d')
        diff = end - start

        return {
            'start_date': start_str,
            'end_date': end_str,
            'days': diff.days,
            'months': round(diff.days / 30, 1),
            'years': round(diff.days / 365, 2),
        }

    def _verify_amount(self, params: dict) -> dict:
        """
        验证金额的合理性

        检查金额数值是否在合理范围内，大小写金额是否一致等。
        """
        amount = float(params.get('amount', 0))
        currency = params.get('currency', 'CNY')

        # 将数字金额转换为中文大写
        chinese_amount = self._number_to_chinese(amount)

        # 如果提供了大写金额，检查是否一致
        provided_chinese = params.get('chinese_amount', '')
        match = True
        if provided_chinese:
            # 简单的一致性检查（去掉空格后比较）
            match = provided_chinese.replace(' ', '') == chinese_amount.replace(' ', '')

        return {
            'amount': amount,
            'currency': currency,
            'chinese_amount': chinese_amount,
            'amount_match': match if provided_chinese else '未提供大写金额进行比对',
        }

    def _number_to_chinese(self, num: float) -> str:
        """
        将数字金额转换为中文大写

        例如: 12345.67 -> 壹万贰仟叁佰肆拾伍元陆角柒分
        """
        digits = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖']
        units = ['', '拾', '佰', '仟']
        big_units = ['', '万', '亿']

        # 分离整数和小数部分
        integer_part = int(num)
        decimal_part = round((num - integer_part) * 100)

        if integer_part == 0:
            result = '零元'
        else:
            result = ''
            str_num = str(integer_part)
            length = len(str_num)

            # 按四位一组处理
            for i, digit in enumerate(str_num):
                pos = length - 1 - i  # 当前位的权重位置
                d = int(digit)
                group_pos = pos % 4  # 组内位置
                big_unit_pos = pos // 4  # 大单位位置

                if d != 0:
                    result += digits[d] + units[group_pos]
                else:
                    # 处理零的显示逻辑
                    if result and not result.endswith('零'):
                        result += '零'

                # 每四位加大单位
                if group_pos == 0 and big_unit_pos > 0:
                    # 去掉末尾的零
                    result = result.rstrip('零')
                    result += big_units[big_unit_pos]

            result = result.rstrip('零') + '元'

        # 处理角和分
        jiao = decimal_part // 10
        fen = decimal_part % 10

        if jiao == 0 and fen == 0:
            result += '整'
        else:
            if jiao > 0:
                result += digits[jiao] + '角'
            if fen > 0:
                result += digits[fen] + '分'

        return result
